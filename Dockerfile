# get slim base image for python
FROM python:3.11.10-bullseye as builder


RUN apt-get update && apt-get install -y libgomp1

COPY ./requirements/requirements.txt /opt/
RUN pip3 install --no-cache-dir -r /opt/requirements.txt
COPY src /opt/src
COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh

WORKDIR /opt/src

RUN chown -R 1000:1000 /opt/src

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/src:${PATH}"
# set non-root user
USER 1000

ENTRYPOINT ["/opt/entry_point.sh"]
