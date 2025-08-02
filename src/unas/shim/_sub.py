import sys
import json

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from unas.shim import build_model, dummy_train, dummy_to_tflite, clear_keras

from unas.searchspace import Model

from unas.compiler.tinyengine.TfliteConvertor import TfliteConvertor
from unas.compiler.tinyengine.GeneralMemoryScheduler import GeneralMemoryScheduler


def main() -> None:
    while True:
        raw = sys.stdin.readline()
        data = json.loads(raw)

        if data == "end":
            break

        net = build_model(Model.from_dict(data["model"]), data["classes"])
        dummy_train(net, data["resolution"])
        tflite = dummy_to_tflite(net, data["resolution"])

        clear_keras()

        converter = TfliteConvertor(tflite)
        converter.parseOperatorInfo()

        out = []
        layer = converter.layer
        memory_scheduler = GeneralMemoryScheduler(
            layer,
            False,
            False,
            outputTables=out,
            inplace=True,
            mem_visual_path=None,
            VisaulizeTrainable=False,
        )
        memory_scheduler.USE_INPLACE = True
        memory_scheduler.allocateMemory()

        flash = memory_scheduler.flash
        sram = memory_scheduler.buffers["input_output"]

        out = {"flash": flash, "sram": sram}
        sys.stdout.write(f"{json.dumps(out)}\n")
        sys.stdout.write(f"end\n")
        sys.stdout.flush()


if __name__ == "__main__":
    from unas.shim.shutup import silence
    silence()

    main()
