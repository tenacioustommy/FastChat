"""
A model worker that executes the model based on vLLM.

See documentations at docs/vllm_integration.md
"""

import argparse
import asyncio
import json
from typing import List, Dict

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from lmdeploy.serve.async_engine import AsyncEngine
from transformers import AutoTokenizer
import uvicorn
from lmdeploy import pipeline, GenerationConfig
from lmdeploy.archs import autoget_backend
import uuid
from fastchat.utils import PickleResponse
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)


app = FastAPI()


class LMDeployWorker(BaseModelWorker):
    model: AsyncEngine

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        conv_template: str,
        multimodal: bool = False,
        no_register: bool = False,
        tp: int = 1,
        max_seq_len: int = 32768,
    ):
        super().__init__(
            controller_addr=controller_addr,
            worker_addr=worker_addr,
            worker_id=worker_id,
            model_path=model_path,
            model_names=model_names,
            limit_worker_concurrency=limit_worker_concurrency,
            conv_template=conv_template,
        )

        backend = autoget_backend(args.model_path)
        if backend == 'pytorch':
            from lmdeploy.messages import PytorchEngineConfig
            backend_config = PytorchEngineConfig(
                tp=tp,
            )
        else:
            from lmdeploy.messages import TurbomindEngineConfig
            backend_config = TurbomindEngineConfig(
                tp=args.tp,
                session_len=self.context_len
            )

        self.model = pipeline(model_path, backend_config=backend_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.context_len = max_seq_len

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: vLLM worker..."
        )
        if not no_register:
            self.init_heart_beat()

        self.session_id_mapping: Dict[int] = {}

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None)


        # if self.tokenizer.eos_token_id is not None:
        #     stop_token_ids.append(self.tokenizer.eos_token_id)
        echo = params.get("echo", True)

        lmdeploy_session_id = self.call_ct
        self.session_id_mapping[request_id] = lmdeploy_session_id

        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        if stop_token_ids is not None:
            for tid in stop_token_ids:
                if tid is not None:
                    s = self.tokenizer.decode(tid)
                    if s != "":
                        stop.add(s)

        gen_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            stop_words=list(stop),
            do_sample=True,
        )

        request = params.get("request", None)

        res = ""
        async for request_output in self.model.generate(
            context,
            gen_config=gen_config,
            session_id=lmdeploy_session_id,
            do_preprocess=False,
        ):
            aborted = False
            if request and await request.is_disconnected():
                await self.abort(request_id)
                # TODO: abort logic
                aborted = True

            res += request_output.response
            ret = {
                "text": res,
                "error_code": 0,
                "usage": {},
                "cumulative_logprob": [],
                "finish_reason": request_output.finish_reason,
            }

            if request_output.finish_reason == "stop":
                yield (json.dumps({**ret, **{"finish_reason": None}}) + "\0").encode()
            yield (json.dumps(ret) + "\0").encode()

            if aborted:
                break

    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode())

    async def abort(self, request_id):
        await self.model.stop_session(self.session_id_mapping[request_id])
        self.session_id_mapping.pop(request_id)


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id):
    # TODO:
    async def abort_request() -> None:
        await worker.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = str(uuid.uuid4().hex)
    params["request_id"] = request_id
    params["request"] = request
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = str(uuid.uuid4().hex)
    params["request_id"] = request_id
    params["request"] = request
    output = await worker.generate(params)
    release_worker_semaphore()
    await worker.abort(request_id)
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return PickleResponse(worker.get_conv_template())


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default=None)
    parser.add_argument(
        "--controller-address", type=str, default=None,
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "-tp", type=int, default=None, help="Tensor parallel size"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=32768, help="Tensor parallel size"
    )

    args = parser.parse_args()
    if args.model_path:
        args.model = args.model_path
    if args.num_gpus > 1:
        args.tensor_parallel_size = args.num_gpus

    import subprocess

    if args.worker_address is None:
        worker_addr = subprocess.getoutput("hostname -I").split()[0]
        worker_addr = f"http://{worker_addr}:{args.port}"
    else:
        worker_addr = args.worker_address
    print(f"worker_addr: {worker_addr}")

    worker = LMDeployWorker(
        controller_addr=args.controller_address,
        worker_addr=worker_addr,
        worker_id=worker_id,
        no_register=args.no_register,
        limit_worker_concurrency=args.limit_worker_concurrency,
        model_path=args.model_path,
        model_names=args.model_names,
        conv_template=args.conv_template,
        tp=args.tp,
        max_seq_len=args.max_seq_len,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")