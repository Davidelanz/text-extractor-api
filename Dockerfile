FROM openvino/ubuntu20_runtime

CMD ["bash"]

# copy repo
WORKDIR /workspace
COPY . .

# install API requirements
USER root
RUN apt-get update & apt-get install python3
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install modelplace-api[vis]
RUN python3 download_models.py

# expose API port
EXPOSE 80

# start API as starting action 
ENTRYPOINT python3 server.py
