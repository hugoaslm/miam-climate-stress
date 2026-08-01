import importlib.util
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def install_miam_stubs():
    if "maskSDM" in sys.modules:
        return

    class GeoPlantDataset:
        pass

    _stub_module("maskSDM")
    _stub_module(
        "maskSDM.config",
        BASE_CONFIG={},
        add_data_specific_parameters=lambda config, data: config,
    )
    _stub_module("maskSDM.data")
    _stub_module(
        "maskSDM.data.geoplant",
        get_geoplant_data=lambda **kw: {},
        filter_species=lambda data, min_num_obs: data,
        split_data=lambda data: data,
        normalize_data=lambda data: data,
        GeoPlantDataset=GeoPlantDataset,
    )
    _stub_module("maskSDM.training")
    _stub_module(
        "maskSDM.training.helpers",
        create_dataloader=lambda *a, **kw: [],
        seed_everything=lambda seed: None,
    )
    _stub_module("maskSDM.modules")
    _stub_module("maskSDM.modules.model", get_model=lambda **kw: None)
    _stub_module("torcheval")
    _stub_module("torcheval.metrics")
    _stub_module(
        "torcheval.metrics.functional",
        binary_auroc=lambda *a, **kw: None,
    )


def import_evaluate_climate_stress():
    install_miam_stubs()
    path = os.path.join(REPO, "scripts", "evaluate_climate_stress.py")
    spec = importlib.util.spec_from_file_location("evaluate_climate_stress", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
