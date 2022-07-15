FROM python:3.8

COPY ./requirements.txt /

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 htop -y

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY cams2 /cams2
COPY dicio /dicio
COPY i3t /i3t

COPY anom /anom
COPY out /out

COPY main.sh /main.sh
COPY train /train
COPY util.py /util.py
COPY monitor.py /monitor.py
COPY decomp.py /decomp.py
COPY simulate_stream.py /simulate_stream.py
COPY denoiser.py /denoiser.py

RUN chmod +x ./main.sh
CMD ./main.sh

