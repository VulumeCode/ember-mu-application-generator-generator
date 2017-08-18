FROM semtech/mu-cl-resources:1.15.0
MAINTAINER Vincent G.

COPY . /app/resources-generator/
ADD ./startup.lisp /usr/src/startup.lisp

CMD sh /load-config.sh; sbcl --load /usr/src/startup.lisp
