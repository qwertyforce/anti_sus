FROM python:3.10-slim-bullseye

ARG USERNAME=app
ARG USER_UID=1000
ARG USER_GID=1000

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY ./ ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt update -y && apt install -y nano screen 
# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME 
RUN chown -R $USER_UID:$USER_GID /app
RUN chmod +x ./wrapper.sh
USER app
CMD ["./wrapper.sh"]