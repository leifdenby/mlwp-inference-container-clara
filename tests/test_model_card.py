"""The model card describes the variables of the trained model."""
import pytest
import yaml

from clara_inference import convert_output
from clara_inference import model_card
from clara_inference import model_configs as mc


def _config_features(fn, category):
    """
    The feature names of `category` in a datastore config, in mllam-data-prep
    order: per input, the variables (per level with the name_format), then
    the derived variables.
    """
    config = yaml.safe_load((model_card.CONFIGS / fn).read_text())
    names = []
    for inp in config["inputs"].values():
        if inp["target_output_variable"] != category:
            continue
        name_format = inp["dim_mapping"][f"{category}_feature"]["name_format"]
        variables = inp.get("variables", [])
        if isinstance(variables, dict):
            for var, dims in variables.items():
                (dim, levels), = dims.items()
                names += [
                    name_format.format(var_name=var, **{dim: level})
                    for level in levels["values"]
                ]
        else:
            names += [name_format.format(var_name=var) for var in variables]
        names += list(inp.get("derived_variables", {}))
    return names


@pytest.fixture(scope="module")
def card():
    return model_card.load()


@pytest.mark.parametrize("group", list(model_card.GROUPS))
def test_variables_are_the_datastore_features(card, group):
    training = mc.training_configs(model_card.CONFIGS)
    datastore, category = model_card.GROUPS[group]
    fn = {
        "interior": training.interior_datastore,
        "boundary": training.boundary_datastore,
    }[datastore]
    assert model_card.features(
        card["groups"][group]["variables"]
    ) == _config_features(fn, category)


def test_state_matches_the_output_files(card):
    state = card["groups"]["state"]["variables"]
    files = {
        "single_levels.zarr": convert_output.SINGLE_LEVEL_VARS,
        "pressure_levels.zarr": convert_output.PRESSURE_LEVEL_VARS,
    }
    for v in state:
        assert v["name"] in files[v["output_file"]], v["name"]
        # the same units, in UDUNITS spelling without "**"
        assert v["units"] == convert_output.UNITS[v["name"]].replace(
            "**", ""
        ), v["name"]
    assert sorted(v["name"] for v in state) == sorted(convert_output.UNITS)


def test_every_state_feature_has_a_loss_weight(card):
    weights = model_card.loss_weights()
    assert set(model_card.features(card["groups"]["state"]["variables"])) == (
        set(weights)
    )


def test_model_card_is_up_to_date(card):
    assert model_card.OUTPUT.read_text() == model_card.render(
        card, model_card.loss_weights()
    ), "regenerate with `python -m clara_inference.model_card`"


def test_load_rejects_inconsistent_cf(tmp_path, card):
    v = card["groups"]["state"]["variables"][0]
    v["cf"] = "no"
    fp = tmp_path / "card.yaml"
    fp.write_text(yaml.safe_dump(card))
    with pytest.raises(ValueError, match="standard_name must be null"):
        model_card.load(fp)
