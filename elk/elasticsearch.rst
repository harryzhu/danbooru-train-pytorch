ElasticSearch
=============

Enable CORS
-----------

Enable cross-domain requests by adding the following to the ``config/elasticsearch.yml`` config file::

    http.cors.enabled:      true 
    http.cors.allow-origin: /.*/  # /http?:\/\/(10\.11\.176\.24)\d+(:[0-9]+)?/ 

Auto-run when OS boot
--------------------

edit ``/etc/init.d/elasticsearch`` ::

    /home/ops/elk/elasticsearch/bin/elasticsearch -d &
    




.. _validate: http://validator.w3.org/
.. _YSlow: http://yslow.org/