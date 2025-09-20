#!/bin/sh
set -e

uvicorn worker:health_app --host 0.0.0.0 --port 8080 &

arq worker.WorkerSettings