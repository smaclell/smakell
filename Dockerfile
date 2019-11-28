from python:3.7

EXPOSE      8080
ENV         APP_HOME /snake
WORKDIR     $APP_HOME

RUN         apt-get update && apt-get install -y \
              cmake \
              libopenmpi-dev \
              python3-dev \
              zlib1g-dev \
            --no-install-recommends && rm -rf /var/lib/apt/lists/*

COPY        requirements.txt $APP_HOME/requirements.txt

RUN         pip install -r $APP_HOME/requirements.txt

COPY        . $APP_HOME
