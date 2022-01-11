import argparse
import os
from loguru import logger
import tensorrt as trt
import torch

# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from mmdet.exp import get_exp
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)



def make_parser():
    parser = argparse.ArgumentParser("MieMieDetection TensorRT")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    return parser


class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32


def build_network(network, weights, exp, args, state_dict):
    from tools_trt.models import Resnet50Vd, Resnet101Vd, Resnet18Vd, YOLOv3Head, PPYOLO
    from tools_trt.models.necks.yolo_fpn import PPYOLOFPN, PPYOLOPAN

    Backbone = None
    if exp.backbone_type == 'Resnet50Vd':
        Backbone = Resnet50Vd
    elif exp.backbone_type == 'Resnet101Vd':
        Backbone = Resnet101Vd
    elif exp.backbone_type == 'Resnet18Vd':
        Backbone = Resnet18Vd
    backbone = Backbone(**exp.backbone)

    # Fpn = None
    # if exp.fpn_type == 'PPYOLOFPN':
    #     Fpn = PPYOLOFPN
    # elif exp.fpn_type == 'PPYOLOPAN':
    #     Fpn = PPYOLOPAN
    # fpn = Fpn(**exp.fpn)
    # head = YOLOv3Head(loss=None, nms_cfg=exp.nms_cfg, **exp.head)
    # model = PPYOLO(backbone, fpn, head)
    model = PPYOLO(backbone, None, None)

    input_tensor = network.add_input(name="image", dtype=trt.float32, shape=(1, 3, 416, 416))
    out = model(input_tensor, network, state_dict)
    # out[0].name = 's16'
    # out[1].name = 's32'
    out.name = 's32'
    network.mark_output(tensor=out)


    # Configure the network layers based on the weights provided.
    # input_tensor = network.add_input(name="image", dtype=trt.float32, shape=(1, 3, 416, 416))
    #
    # conv1_w = np.random.randn(20, 1, 5, 5)
    # conv1_b = np.random.randn(20, )
    # conv1 = network.add_convolution(input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w, bias=conv1_b)
    # conv1.stride = (1, 1)
    # conv1.padding = (2, 2)
    #
    # pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    # pool1.stride = (2, 2)
    #
    # conv2_w = weights['conv2.weight'].numpy()
    # conv2_b = weights['conv2.bias'].numpy()
    # conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
    # conv2.stride = (1, 1)
    #
    # pool2 = network.add_pooling(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
    # pool2.stride = (2, 2)
    #
    # fc1_w = weights['fc1.weight'].numpy()
    # fc1_b = weights['fc1.bias'].numpy()
    # fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)
    #
    # relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)
    #
    # fc2_w = weights['fc2.weight'].numpy()
    # fc2_b = weights['fc2.bias'].numpy()
    # fc2 = network.add_fully_connected(relu1.get_output(0), ModelData.OUTPUT_SIZE, fc2_w, fc2_b)
    #
    # fc2.get_output(0).name = ModelData.OUTPUT_NAME
    # network.mark_output(tensor=fc2.get_output(0))


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream



# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


@logger.catch
def main(exp, args):
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["model"]
    # For more information on TRT basics, refer to the introductory samples.
    builder = trt.Builder(TRT_LOGGER)
    aaaaaaa = EXPLICIT_BATCH
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    runtime = trt.Runtime(TRT_LOGGER)

    config.max_workspace_size = GiB(1)
    # Populate the network using weights from the PyTorch model.
    build_network(network, state_dict, exp, args, state_dict)
    # Build and return an engine.
    plan = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(plan)

    # Build an engine, allocate buffers and create a stream.
    # For more information on buffer allocation, refer to the introductory samples.
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()


    dic2 = np.load('../tools/data.npz')
    img = dic2['img']
    aaa1 = dic2['aaa1']
    aaa2 = dic2['aaa2']
    aaa3 = dic2['aaa3']
    pool = dic2['pool']
    stage2_0 = dic2['stage2_0']
    s8 = dic2['s8']
    s16 = dic2['s16']
    s32 = dic2['s32']
    img = img.ravel().astype(np.float32)

    # she zhi shu ru.
    np.copyto(inputs[0].host, img)

    # For more information on performing inference, refer to the introductory samples.
    # The common.do_inference function will return a list of outputs - we only have one in this case.
    [output] = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    # output = np.reshape(output, (1, 32, 208, 208))
    output = np.reshape(output, s8.shape)
    ddd = np.sum((output - s8) ** 2)
    print('ddd=%.6f' % ddd)
    print()




if __name__ == "__main__":
    args = make_parser().parse_args()
    # 判断是否是调试状态
    isDebug = True if sys.gettrace() else False
    if isDebug:
        print('Debug Mode.')
        args.exp_file = '../' + args.exp_file
        args.ckpt = '../' + args.ckpt   # 如果是绝对路径，把这一行注释掉
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)










