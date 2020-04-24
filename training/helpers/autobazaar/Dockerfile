FROM python:3.6-buster

RUN mkdir /autobazaar && \
    mkdir /abz && \
    ln -s /input /abz/input && \
    ln -s /output /abz/output

# Copy code
COPY setup.py README.md HISTORY.md MANIFEST.in /autobazaar/

# Install project
RUN pip3 install -e /autobazaar && pip install ipdb

COPY autobazaar /autobazaar/autobazaar

WORKDIR /abz

CMD ["echo", "Usage: docker run -ti -u$UID -v $(pwd):/abz mlbazaar/autobazaar abz OPTIONS"]
