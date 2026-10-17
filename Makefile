# 常用命令速查：make help
.PHONY: help init build up down restart logs test ingest eval clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

init:  ## 生成 .env（不存在时才创建）
	cp -n .env.example .env || true

build:  ## 构建镜像
	docker compose build

up:     ## 启动（数据库 + 服务）
	docker compose up -d

down:   ## 停止
	docker compose down

restart: ## 重启服务（改完代码用这个）
	docker compose restart app

logs:   ## 查看服务日志
	docker compose logs -f app

test:   ## 单元测试
	python -m pytest -q

ingest: ## 索引文档：make ingest p=./docs c=hr
	python -m scripts.ingest $(p) --collection $(or $(c),default) --init-db

eval:   ## 评测：make eval c=hr s=20
	python -m scripts.evaluate --collection $(or $(c),default) --size $(or $(s),20)

clean:  ## 清空数据与上传文件（危险）
	docker compose down -v && rm -rf data/uploads/* data/eval/*
