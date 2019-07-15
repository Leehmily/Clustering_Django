FROM python:3.6
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY pip.conf /root/.pip/pip.conf
COPY requirements.txt /usr/src/app/
RUN pip install -r /usr/src/app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN rm -rf /usr/src/app
COPY . /usr/src/app
CMD [ "python", "./manage.py", "runserver", "0.0.0.0:8080"]
sudo