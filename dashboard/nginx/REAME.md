# Important: must match nginx structure

By default, nginx expects content to live in `/etc/nginx`.
So this directory must match the directory structure under `/etc/nginx` in order
for the dockerfile to setup things correctly.

For example:

This repo's `nginx/conf.d` will map to `/etc/nginx/conf.d`