FROM tensorflow/tensorflow:2.0.0rc0-gpu-py3-jupyter

RUN apt update
RUN apt install -y vim git
RUN pip install jupyterlab
# tqdm
RUN pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension
RUN curl -sL https://deb.nodesource.com/setup_11.x | bash -
RUN apt install -y nodejs
RUN pip install tqdm
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN pip install sklearn yapf

EXPOSE 6006
EXPOSE 8888
EXPOSE 5000

WORKDIR /

