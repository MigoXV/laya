"""以项目默认监听地址启动 Demo。"""

import logging

import typer

from examples.demo01.app import DemoConfig, create_app


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
app = typer.Typer()


@app.callback()
def main():
    """Laya 可视化推理 Demo。"""


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", envvar="LAYA_DEMO_HOST"),
    port: int = typer.Option(10013, envvar="LAYA_DEMO_PORT"),
):
    import uvicorn

    config = DemoConfig(host=host, port=port)
    uvicorn.run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    app()
