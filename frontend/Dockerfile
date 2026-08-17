FROM nginx:alpine

# Copy frontend files into nginx
COPY . /usr/share/nginx/html

# Change nginx from port 80 to Railway's expected port
RUN sed -i 's/listen       80;/listen       8080;/' /etc/nginx/conf.d/default.conf

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]