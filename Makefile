IMAGE = 1r5agbgaofhieysdt9esr/ucdavis_folic:latest

build:
	docker build --platform linux/amd64 -t $(IMAGE) .

run:
	docker run --rm -it --platform linux/amd64 $(IMAGE)

push:
	docker push $(IMAGE)

go:
	make build && make push