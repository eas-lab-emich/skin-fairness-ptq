"""
Energy simulator — 6 architectures, mixed-precision quantization.

VGG11, ResNet18, ResNet34, ResNet50:
  nn_dataflow scheduler (run once for W8A8 reference, then analytically scale
  per-layer MAC cost by (wbits*abits)/(8*8); memory/NoC/static costs unchanged).

MobileNetV2, ShuffleNet V2:
  Analytical MAC model only — nn_dataflow doesn't support depthwise/grouped convs.

Outputs:
  search/results/energy_results.csv
  search/results/energy_results.xlsx
    Sheet "All Methods"              : 6 archs x 5 methods
    Sheet "NSGA-II Pareto (ResNet50)": 50 Pareto solutions
"""

import sys, os, json, re
import torch, torchvision
from torch.nn import Conv2d, Linear
import pandas as pd

sys.path.insert(0, '/home/mussa/nn_dataflow')

from nn_dataflow.core import NNDataflow, Cost, Option, Resource
from nn_dataflow.core import MapStrategyEyeriss, NodeRegion, PhyDim2
from nn_dataflow.core import MemHierEnum as me
from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, PoolingLayer, EltwiseLayer
from nn_dataflow.nns import resnet50 as resnet50_nn

RESULTS     = '/home/mussa/skin-fairness-ptq/search/results'
BA_FILE     = os.path.join(RESULTS, 'bit_assignments.json')
PARETO_FILES = {
    'vgg11'             : os.path.join(RESULTS, 'nsga2_vgg11_s42_final_paper/pareto_front.json'),
    'resnet18'          : os.path.join(RESULTS, 'nsga2_resnet18_s42_final_paper/pareto_front.json'),
    'resnet34'          : os.path.join(RESULTS, 'nsga2_resnet34_s42_final_paper/pareto_front.json'),
    'resnet50'          : os.path.join(RESULTS, 'nsga2_resnet50_s42_final_paper/pareto_front.json'),
    'mobilenet_v2'      : os.path.join(RESULTS, 'nsga2_mobilenet_v2_s42_final_paper/pareto_front.json'),
    'shufflenet_v2_x1_0': os.path.join(RESULTS, 'nsga2_shufflenet_v2_x1_0_s42_final_paper/pareto_front.json'),
}
ACC_CSV     = os.path.join(RESULTS, 'final_comparison_final_paper.csv')
OUT_CSV     = os.path.join(RESULTS, 'energy_results_final_paper.csv')
OUT_XLSX    = os.path.join(RESULTS, 'energy_results_final_paper.xlsx')

# ---------------------------------------------------------------------------
# nn_dataflow hardware config (from professor's example command)
# --batch 16 --word 8 --array 12 14 --nodes 1 1
# --regf 168 --gbuf 55296 --bus-width 64 --disable-interlayer-opt
# ---------------------------------------------------------------------------
BATCH      = 16
WORD_BYTES = 1
DIM_NODES  = PhyDim2(1, 1)
DIM_ARRAY  = PhyDim2(12, 14)
SIZE_GBUF  = 55296 // WORD_BYTES
SIZE_REGF  = 168   // WORD_BYTES
BUS_WIDTH  = 8      # 64 bits / 8 bits-per-word

proc_region = NodeRegion(dim=DIM_NODES, origin=PhyDim2(0,0), type=NodeRegion.PROC)
data_region = NodeRegion(dim=DIM_NODES, origin=PhyDim2(0,0), type=NodeRegion.DRAM)

resource = Resource(proc_region=proc_region,
                    dram_region=data_region,
                    src_data_region=data_region,
                    dst_data_region=data_region,
                    dim_array=DIM_ARRAY,
                    size_gbuf=SIZE_GBUF,
                    size_regf=SIZE_REGF,
                    array_bus_width=BUS_WIDTH,
                    dram_bandwidth=float('inf'),
                    no_time_mux=False)

hier_cost = [0] * me.NUM
hier_cost[me.DRAM] = 200
hier_cost[me.GBUF] = 6
hier_cost[me.ITCN] = 2
hier_cost[me.REGF] = 1
cost = Cost(mac_op=1, mem_hier=tuple(hier_cost), noc_hop=10, idl_unit=0)

options = Option(sw_gbuf_bypass=(True, True, True),
                 sw_solve_loopblocking=False,
                 hw_access_forwarding=False,
                 hw_gbuf_sharing=False,
                 hw_gbuf_save_writeback=False,
                 partition_hybrid=False,
                 partition_batch=False,
                 partition_ifmaps=False,
                 partition_interlayer=False,
                 layer_pipeline_time_ovhd=0,
                 layer_pipeline_max_degree=float('inf'),
                 layer_pipeline_opt=False,
                 opt_goal='e',
                 ntops=1,
                 nprocesses=1,
                 verbose=False)

# ---------------------------------------------------------------------------
# nn_dataflow network builders (layer names match bit_assignments.json keys)
# ---------------------------------------------------------------------------
def _c(name, wt_bits, act_bits, nifm, nofm, sofm, sfil, strd=1):
    wb = wt_bits.get(name, 8)
    ab = act_bits.get(name, 8)
    return ConvLayer(nifm, nofm, sofm, sfil, strd=strd, wbits=wb, abits=ab)

def _f(name, wt_bits, act_bits, nifm, nofm, sfil=1):
    wb = wt_bits.get(name, 8)
    ab = act_bits.get(name, 8)
    return FCLayer(nifm, nofm, sfil, wbits=wb, abits=ab)

def build_vgg11(wt_bits=None, act_bits=None):
    wt_bits = wt_bits or {}; act_bits = act_bits or {}
    net = Network('VGG11')
    net.set_input_layer(InputLayer(3, 224))
    net.add('features.0',  _c('features.0',  wt_bits, act_bits, 3,   64,  224, 3))
    net.add('pool1',       PoolingLayer(64,  112, 2))
    net.add('features.3',  _c('features.3',  wt_bits, act_bits, 64,  128, 112, 3))
    net.add('pool2',       PoolingLayer(128,  56, 2))
    net.add('features.6',  _c('features.6',  wt_bits, act_bits, 128, 256,  56, 3))
    net.add('features.8',  _c('features.8',  wt_bits, act_bits, 256, 256,  56, 3))
    net.add('pool3',       PoolingLayer(256,  28, 2))
    net.add('features.11', _c('features.11', wt_bits, act_bits, 256, 512,  28, 3))
    net.add('features.13', _c('features.13', wt_bits, act_bits, 512, 512,  28, 3))
    net.add('pool4',       PoolingLayer(512,  14, 2))
    net.add('features.16', _c('features.16', wt_bits, act_bits, 512, 512,  14, 3))
    net.add('features.18', _c('features.18', wt_bits, act_bits, 512, 512,  14, 3))
    net.add('pool5',       PoolingLayer(512,   7, 2))
    net.add('classifier.0', _f('classifier.0', wt_bits, act_bits, 512,  4096, sfil=7))
    net.add('classifier.3', _f('classifier.3', wt_bits, act_bits, 4096, 4096))
    net.add('classifier.6', _f('classifier.6', wt_bits, act_bits, 4096, 9))
    return net

def build_resnet18(wt_bits=None, act_bits=None):
    wt_bits = wt_bits or {}; act_bits = act_bits or {}
    net = Network('ResNet18')
    net.set_input_layer(InputLayer(3, 224))
    net.add('conv1',  _c('conv1', wt_bits, act_bits, 3, 64, 112, 7, strd=2))
    net.add('pool1',  PoolingLayer(64, 56, 3, strd=2))
    # layer1 — no downsample, skip from pool1 / previous res
    net.add('layer1.0.conv1', _c('layer1.0.conv1', wt_bits, act_bits, 64, 64, 56, 3))
    net.add('layer1.0.conv2', _c('layer1.0.conv2', wt_bits, act_bits, 64, 64, 56, 3))
    net.add('layer1.0.res',   EltwiseLayer(64, 56, 2), prevs=('pool1', 'layer1.0.conv2'))
    net.add('layer1.1.conv1', _c('layer1.1.conv1', wt_bits, act_bits, 64, 64, 56, 3))
    net.add('layer1.1.conv2', _c('layer1.1.conv2', wt_bits, act_bits, 64, 64, 56, 3))
    net.add('layer1.1.res',   EltwiseLayer(64, 56, 2), prevs=('layer1.0.res', 'layer1.1.conv2'))
    # layer2 — first block has downsample
    net.add('layer2.0.conv1',        _c('layer2.0.conv1',        wt_bits, act_bits, 64,  128, 28, 3, strd=2))
    net.add('layer2.0.conv2',        _c('layer2.0.conv2',        wt_bits, act_bits, 128, 128, 28, 3))
    net.add('layer2.0.downsample.0', _c('layer2.0.downsample.0', wt_bits, act_bits, 64,  128, 28, 1, strd=2), prevs=('layer1.1.res',))
    net.add('layer2.0.res', EltwiseLayer(128, 28, 2), prevs=('layer2.0.downsample.0', 'layer2.0.conv2'))
    net.add('layer2.1.conv1', _c('layer2.1.conv1', wt_bits, act_bits, 128, 128, 28, 3))
    net.add('layer2.1.conv2', _c('layer2.1.conv2', wt_bits, act_bits, 128, 128, 28, 3))
    net.add('layer2.1.res',   EltwiseLayer(128, 28, 2), prevs=('layer2.0.res', 'layer2.1.conv2'))
    # layer3
    net.add('layer3.0.conv1',        _c('layer3.0.conv1',        wt_bits, act_bits, 128, 256, 14, 3, strd=2))
    net.add('layer3.0.conv2',        _c('layer3.0.conv2',        wt_bits, act_bits, 256, 256, 14, 3))
    net.add('layer3.0.downsample.0', _c('layer3.0.downsample.0', wt_bits, act_bits, 128, 256, 14, 1, strd=2), prevs=('layer2.1.res',))
    net.add('layer3.0.res', EltwiseLayer(256, 14, 2), prevs=('layer3.0.downsample.0', 'layer3.0.conv2'))
    net.add('layer3.1.conv1', _c('layer3.1.conv1', wt_bits, act_bits, 256, 256, 14, 3))
    net.add('layer3.1.conv2', _c('layer3.1.conv2', wt_bits, act_bits, 256, 256, 14, 3))
    net.add('layer3.1.res',   EltwiseLayer(256, 14, 2), prevs=('layer3.0.res', 'layer3.1.conv2'))
    # layer4
    net.add('layer4.0.conv1',        _c('layer4.0.conv1',        wt_bits, act_bits, 256, 512, 7, 3, strd=2))
    net.add('layer4.0.conv2',        _c('layer4.0.conv2',        wt_bits, act_bits, 512, 512, 7, 3))
    net.add('layer4.0.downsample.0', _c('layer4.0.downsample.0', wt_bits, act_bits, 256, 512, 7, 1, strd=2), prevs=('layer3.1.res',))
    net.add('layer4.0.res', EltwiseLayer(512, 7, 2), prevs=('layer4.0.downsample.0', 'layer4.0.conv2'))
    net.add('layer4.1.conv1', _c('layer4.1.conv1', wt_bits, act_bits, 512, 512, 7, 3))
    net.add('layer4.1.conv2', _c('layer4.1.conv2', wt_bits, act_bits, 512, 512, 7, 3))
    net.add('layer4.1.res',   EltwiseLayer(512, 7, 2), prevs=('layer4.0.res', 'layer4.1.conv2'))
    net.add('avgpool', PoolingLayer(512, 1, 7))
    net.add('fc', _f('fc', wt_bits, act_bits, 512, 9))
    return net

def build_resnet34(wt_bits=None, act_bits=None):
    wt_bits = wt_bits or {}; act_bits = act_bits or {}
    net = Network('ResNet34')
    net.set_input_layer(InputLayer(3, 224))
    net.add('conv1', _c('conv1', wt_bits, act_bits, 3, 64, 112, 7, strd=2))
    net.add('pool1', PoolingLayer(64, 56, 3, strd=2))

    prev = 'pool1'
    # layer1 — 3 basic blocks, 64ch, 56x56 (no downsample; skip from pool1/prev res)
    for i in range(3):
        skip = prev
        net.add(f'layer1.{i}.conv1', _c(f'layer1.{i}.conv1', wt_bits, act_bits, 64, 64, 56, 3))
        net.add(f'layer1.{i}.conv2', _c(f'layer1.{i}.conv2', wt_bits, act_bits, 64, 64, 56, 3))
        net.add(f'layer1.{i}.res',   EltwiseLayer(64, 56, 2), prevs=(skip, f'layer1.{i}.conv2'))
        prev = f'layer1.{i}.res'

    # layer2 — 4 basic blocks, 128ch, 28x28 (first has downsample)
    net.add('layer2.0.conv1',        _c('layer2.0.conv1',        wt_bits, act_bits, 64,  128, 28, 3, strd=2))
    net.add('layer2.0.conv2',        _c('layer2.0.conv2',        wt_bits, act_bits, 128, 128, 28, 3))
    net.add('layer2.0.downsample.0', _c('layer2.0.downsample.0', wt_bits, act_bits, 64,  128, 28, 1, strd=2), prevs=(prev,))
    net.add('layer2.0.res', EltwiseLayer(128, 28, 2), prevs=('layer2.0.downsample.0', 'layer2.0.conv2'))
    prev = 'layer2.0.res'
    for i in range(1, 4):
        skip = prev
        net.add(f'layer2.{i}.conv1', _c(f'layer2.{i}.conv1', wt_bits, act_bits, 128, 128, 28, 3))
        net.add(f'layer2.{i}.conv2', _c(f'layer2.{i}.conv2', wt_bits, act_bits, 128, 128, 28, 3))
        net.add(f'layer2.{i}.res',   EltwiseLayer(128, 28, 2), prevs=(skip, f'layer2.{i}.conv2'))
        prev = f'layer2.{i}.res'

    # layer3 — 6 basic blocks, 256ch, 14x14
    net.add('layer3.0.conv1',        _c('layer3.0.conv1',        wt_bits, act_bits, 128, 256, 14, 3, strd=2))
    net.add('layer3.0.conv2',        _c('layer3.0.conv2',        wt_bits, act_bits, 256, 256, 14, 3))
    net.add('layer3.0.downsample.0', _c('layer3.0.downsample.0', wt_bits, act_bits, 128, 256, 14, 1, strd=2), prevs=(prev,))
    net.add('layer3.0.res', EltwiseLayer(256, 14, 2), prevs=('layer3.0.downsample.0', 'layer3.0.conv2'))
    prev = 'layer3.0.res'
    for i in range(1, 6):
        skip = prev
        net.add(f'layer3.{i}.conv1', _c(f'layer3.{i}.conv1', wt_bits, act_bits, 256, 256, 14, 3))
        net.add(f'layer3.{i}.conv2', _c(f'layer3.{i}.conv2', wt_bits, act_bits, 256, 256, 14, 3))
        net.add(f'layer3.{i}.res',   EltwiseLayer(256, 14, 2), prevs=(skip, f'layer3.{i}.conv2'))
        prev = f'layer3.{i}.res'

    # layer4 — 3 basic blocks, 512ch, 7x7
    net.add('layer4.0.conv1',        _c('layer4.0.conv1',        wt_bits, act_bits, 256, 512, 7, 3, strd=2))
    net.add('layer4.0.conv2',        _c('layer4.0.conv2',        wt_bits, act_bits, 512, 512, 7, 3))
    net.add('layer4.0.downsample.0', _c('layer4.0.downsample.0', wt_bits, act_bits, 256, 512, 7, 1, strd=2), prevs=(prev,))
    net.add('layer4.0.res', EltwiseLayer(512, 7, 2), prevs=('layer4.0.downsample.0', 'layer4.0.conv2'))
    prev = 'layer4.0.res'
    for i in range(1, 3):
        skip = prev
        net.add(f'layer4.{i}.conv1', _c(f'layer4.{i}.conv1', wt_bits, act_bits, 512, 512, 7, 3))
        net.add(f'layer4.{i}.conv2', _c(f'layer4.{i}.conv2', wt_bits, act_bits, 512, 512, 7, 3))
        net.add(f'layer4.{i}.res',   EltwiseLayer(512, 7, 2), prevs=(skip, f'layer4.{i}.conv2'))
        prev = f'layer4.{i}.res'

    net.add('avgpool', PoolingLayer(512, 1, 7))
    net.add('fc', _f('fc', wt_bits, act_bits, 512, 9))
    return net

# ResNet50 ba_key mapping (nn_dataflow internal names → PyTorch names)
def _resnet50_ba_key(nn_name):
    if nn_name in ('conv1', 'fc'):
        return nn_name
    m = re.match(r'conv(\d)_(\d+)_([abc])$', nn_name)
    if m:
        stage, block, letter = m.groups()
        return 'layer{}.{}.conv{}'.format(int(stage)-1, block, {'a':1,'b':2,'c':3}[letter])
    m = re.match(r'conv(\d)_br$', nn_name)
    if m:
        return 'layer{}.0.downsample.0'.format(int(m.group(1))-1)
    return None

# ---------------------------------------------------------------------------
# MACs for analytical model (MobileNetV2, ShuffleNet V2)
# ---------------------------------------------------------------------------
def get_macs(model):
    macs = {}
    hooks = []
    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(module, Conv2d):
                _, c_out, h_out, w_out = out.shape
                k = module.kernel_size[0] * module.kernel_size[1]
                macs[name] = int(module.in_channels * c_out * h_out * w_out * k / module.groups)
            elif isinstance(module, Linear):
                macs[name] = int(module.in_features * module.out_features)
        return hook
    for name, mod in model.named_modules():
        if isinstance(mod, (Conv2d, Linear)):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 224, 224))
    for h in hooks: h.remove()
    return macs

def analytical_energy(wt_bits, act_bits, macs):
    return sum(wt_bits.get(l, 8) * act_bits.get(l, 8) * n for l, n in macs.items())

# ---------------------------------------------------------------------------
# Run nn_dataflow W8A8 reference and extract per-layer cost breakdown
# ---------------------------------------------------------------------------
def run_reference(network, arch_label):
    print(f'  Running nn_dataflow W8A8 reference for {arch_label}...')
    nnd = NNDataflow(network, BATCH, resource, cost, MapStrategyEyeriss)
    tops, _ = nnd.schedule_search(options)
    assert tops, f'No valid schedule found for {arch_label}'
    ref = tops[0]
    print(f'    total_cost = {ref.total_cost:.2e}')
    return ref

def energy_from_ref(ref, wt_bits, act_bits, ba_key_fn=None):
    """Scale per-layer MAC cost by bit-width; keep memory/NoC/static from W8A8."""
    sum_cost   = 0.0
    sum_static = 0.0
    for nn_name, sr in ref.res_dict.items():
        key = ba_key_fn(nn_name) if ba_key_fn else nn_name
        wb  = wt_bits.get(key,  8) if key else 8
        ab  = act_bits.get(key, 8) if key else 8
        new_cost_op = sr.scheme['cost_op'] * (wb * ab) / 64.0
        sum_cost   += new_cost_op + sr.scheme['cost_access'] + \
                      sr.scheme['cost_noc'] + sr.scheme['cost_static']
        sum_static += sr.scheme['cost_static']

    ref_total_time = ref.total_time
    ref_sum_time   = sum(sr.total_time for sr in ref.res_dict.values())
    if ref_sum_time > 0:
        overcounted = sum_static * (1.0 - ref_total_time / ref_sum_time)
        return sum_cost - overcounted
    return sum_cost

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(BA_FILE) as f:  ba     = json.load(f)

acc_df = pd.read_csv(ACC_CSV)
acc_lookup = {}
for _, row in acc_df.iterrows():
    acc_lookup[(str(row['arch']).strip(), str(row['method']).strip())] = row

METHOD_KEY_TO_CSV = {
    'fp32'                 : 'FP32',
    'w8a8_uniform'         : 'W8A8 (uniform)',
    'w6a6_uniform'         : 'W6A6 (uniform)',
    'greedy_unconstrained' : 'Greedy W6A6 (unconstrained)',
    'greedy_ud_constrained': 'Greedy W6A6 (UD-constrained)',
}
ARCH_DISPLAY = {
    'vgg11'              : 'VGG11',
    'resnet18'           : 'ResNet18',
    'resnet34'           : 'ResNet34',
    'resnet50'           : 'ResNet50',
    'mobilenet_v2'       : 'MobileNetV2',
    'shufflenet_v2_x1_0' : 'ShuffleNet V2',
}

# ---------------------------------------------------------------------------
# Per-architecture energy computation
# ---------------------------------------------------------------------------
NN_ARCHS  = ['vgg11', 'resnet18', 'resnet34', 'resnet50']
MAC_ARCHS = ['mobilenet_v2', 'shufflenet_v2_x1_0']

BUILDERS = {
    'vgg11'   : build_vgg11,
    'resnet18': build_resnet18,
    'resnet34': build_resnet34,
    'resnet50': lambda wt, ab: resnet50_nn.build(wt, ab, num_classes=9),
}
BA_KEY_FNS = {
    'vgg11'   : None,
    'resnet18': None,
    'resnet34': None,
    'resnet50': _resnet50_ba_key,
}

# Build MACs for analytical architectures
print('Computing MACs for MobileNetV2 and ShuffleNet V2...')
mv2 = torchvision.models.mobilenet_v2(weights=None)
mv2.classifier[1] = Linear(1280, 9)
sn2 = torchvision.models.shufflenet_v2_x1_0(weights=None)
sn2.fc = Linear(1024, 9)
MAC_DB = {
    'mobilenet_v2'       : get_macs(mv2),
    'shufflenet_v2_x1_0' : get_macs(sn2),
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
method_rows = []
nn_refs     = {}  # arch → {ref, ba_key_fn, e_fp32, e_w8a8}
mac_refs    = {}  # arch → {macs, e_fp32, e_w8a8}

for arch in ARCH_DISPLAY:
    arch_ba  = ba[arch]
    label    = ARCH_DISPLAY[arch]
    print(f'\n=== {label} ===')

    if arch in NN_ARCHS:
        # Run nn_dataflow reference (W8A8)
        wt_8 = arch_ba['w8a8_uniform']['weights']
        ab_8 = arch_ba['w8a8_uniform']['activations']
        ref  = run_reference(BUILDERS[arch](wt_8, ab_8), label)
        ba_key_fn = BA_KEY_FNS[arch]

        # Reference energies for normalisation
        e_fp32  = energy_from_ref(ref, arch_ba['fp32']['weights'],
                                       arch_ba['fp32']['activations'], ba_key_fn)
        e_w8a8  = energy_from_ref(ref, wt_8, ab_8, ba_key_fn)

        nn_refs[arch] = {
            'ref': ref, 'ba_key_fn': ba_key_fn,
            'e_fp32': e_fp32, 'e_w8a8': e_w8a8,
        }

        def get_energy(mk):
            return energy_from_ref(ref, arch_ba[mk]['weights'],
                                        arch_ba[mk]['activations'], ba_key_fn)
    else:
        macs   = MAC_DB[arch]
        e_fp32 = analytical_energy(arch_ba['fp32']['weights'],
                                   arch_ba['fp32']['activations'], macs)
        e_w8a8 = analytical_energy(arch_ba['w8a8_uniform']['weights'],
                                   arch_ba['w8a8_uniform']['activations'], macs)
        mac_refs[arch] = {'macs': macs, 'e_fp32': e_fp32, 'e_w8a8': e_w8a8}
        def get_energy(mk):
            return analytical_energy(arch_ba[mk]['weights'],
                                     arch_ba[mk]['activations'], macs)

    for mk, ml in METHOD_KEY_TO_CSV.items():
        energy = get_energy(mk)
        s = acc_lookup.get((arch, ml), {})

        def safe(col):
            v = s.get(col) if isinstance(s, dict) else (s[col] if col in s.index else None)
            try: return round(float(v), 4) if v == v and v is not None else None
            except: return None

        row = {
            'Architecture'               : label,
            'Method'                     : ml,
            'Energy Model'               : 'nn_dataflow' if arch in NN_ARCHS else 'MAC analytical',
            'top1 (%)'                   : safe('top1'),
            'WGA'                        : safe('wga'),
            'UD'                         : safe('ud'),
            'macro_F1'                   : safe('macro_f1'),
            'Avg Wt Bits'                : safe('avg_wt_bits'),
            'Energy (raw)'               : int(energy),
            'Energy vs FP32 (%)'         : round(energy / e_fp32 * 100, 2),
            'Energy vs W8A8 (%)'         : round(energy / e_w8a8 * 100, 2),
            'Energy Savings vs FP32 (%)' : round((1 - energy / e_fp32) * 100, 2),
            'Energy Savings vs W8A8 (%)' : round((1 - energy / e_w8a8) * 100, 2),
        }
        method_rows.append(row)
        print(f'  {ml:<38} E/W8A8={row["Energy vs W8A8 (%)"]:6.1f}%  '
              f'savings={row["Energy Savings vs W8A8 (%)"]:5.1f}%')

# ---------------------------------------------------------------------------
# NSGA-II Pareto fronts (nn_dataflow archs)
# ---------------------------------------------------------------------------
pareto_sheets = {}
for arch in NN_ARCHS:
    pf_path = PARETO_FILES.get(arch)
    label   = ARCH_DISPLAY[arch]
    if not pf_path or not os.path.exists(pf_path):
        print(f'\n=== NSGA-II Pareto ({label}) — SKIPPED (no pareto_front.json) ===')
        continue
    print(f'\n=== NSGA-II Pareto ({label}) ===')
    with open(pf_path) as f:
        pareto = json.load(f)
    pareto.sort(key=lambda x: -x['top1'])

    info = nn_refs[arch]
    rows = []
    for i, sol in enumerate(pareto):
        energy = energy_from_ref(info['ref'], sol['assignment_wts'],
                                 sol['assignment_acts'], info['ba_key_fn'])
        rows.append({
            'Rank (by top1)'             : i + 1,
            'top1 (%)'                   : round(sol['top1'],     2),
            'UD'                         : round(sol['ud'],       4),
            'Avg Bits'                   : round(sol['avg_bits'], 2),
            'Energy (raw)'               : int(energy),
            'Energy vs FP32 (%)'         : round(energy / info['e_fp32'] * 100, 2),
            'Energy vs W8A8 (%)'         : round(energy / info['e_w8a8'] * 100, 2),
            'Energy Savings vs FP32 (%)' : round((1 - energy / info['e_fp32']) * 100, 2),
            'Energy Savings vs W8A8 (%)' : round((1 - energy / info['e_w8a8']) * 100, 2),
        })
    pareto_sheets[label] = pd.DataFrame(rows)
    print(f'  Done — {len(rows)} Pareto solutions computed')

# Pareto fronts for analytical-MAC archs (MobileNetV2, ShuffleNet V2)
for arch in MAC_ARCHS:
    pf_path = PARETO_FILES.get(arch)
    label   = ARCH_DISPLAY[arch]
    if not pf_path or not os.path.exists(pf_path):
        print(f'\n=== NSGA-II Pareto ({label}) — SKIPPED (no pareto_front.json) ===')
        continue
    print(f'\n=== NSGA-II Pareto ({label}) — analytical MAC ===')
    with open(pf_path) as f:
        pareto = json.load(f)
    pareto.sort(key=lambda x: -x['top1'])

    info = mac_refs[arch]
    rows = []
    for i, sol in enumerate(pareto):
        energy = analytical_energy(sol['assignment_wts'], sol['assignment_acts'], info['macs'])
        rows.append({
            'Rank (by top1)'             : i + 1,
            'top1 (%)'                   : round(sol['top1'],     2),
            'UD'                         : round(sol['ud'],       4),
            'Avg Bits'                   : round(sol['avg_bits'], 2),
            'Energy (raw)'               : int(energy),
            'Energy vs FP32 (%)'         : round(energy / info['e_fp32'] * 100, 2),
            'Energy vs W8A8 (%)'         : round(energy / info['e_w8a8'] * 100, 2),
            'Energy Savings vs FP32 (%)' : round((1 - energy / info['e_fp32']) * 100, 2),
            'Energy Savings vs W8A8 (%)' : round((1 - energy / info['e_w8a8']) * 100, 2),
        })
    pareto_sheets[label] = pd.DataFrame(rows)
    print(f'  Done — {len(rows)} Pareto solutions computed')

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
df_methods = pd.DataFrame(method_rows)
df_methods.to_csv(OUT_CSV, index=False)
with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as writer:
    df_methods.to_excel(writer, sheet_name='All Methods', index=False)
    for label, df in pareto_sheets.items():
        df.to_excel(writer, sheet_name=f'NSGA-II Pareto ({label})', index=False)

print(f'\nDone! Excel: {OUT_XLSX}')
