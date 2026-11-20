# Usage: .\run_docker.ps1 <ids...>   One container per model: --cpus=1 --memory=1g, models + input mounted read-only.
$env:Path += ';C:\Users\ASUS\AppData\Local\Programs\DockerDesktop\resources\bin'
docker build -t iot-bench .
python run_bench_docker.py @args
