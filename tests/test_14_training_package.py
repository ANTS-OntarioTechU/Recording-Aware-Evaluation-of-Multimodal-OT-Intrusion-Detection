"""Training-release checks only. No estimator fitting, real-data extraction or downloads."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training.inputs import input_inventory, verify_source
from src.training.workspace import TrainingWorkspace, sha256_file


def _definitions(path):
    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for child in ast.walk(target):
                    if isinstance(child, ast.Name):
                        found[child.id] = node
    return found


def test_original_definitions_keep_recorded_ast_hashes(root):
    manifest = json.loads((root/'configs/training_source_manifest.json').read_text())
    checked = 0
    for source in manifest['sources']:
        definitions = _definitions(root/source['packaged_path'])
        for name, expected in source['definitions'].items():
            actual = hashlib.sha256(ast.dump(definitions[name], include_attributes=False).encode()).hexdigest()
            assert actual == expected, (source['packaged_path'], name)
            checked += 1
    assert checked >= 40
    assert manifest['not_end_to_end_verified'] is True


def test_all_training_modules_import_without_fitting(root, monkeypatch):
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    def forbidden(*args, **kwargs):
        raise AssertionError('An estimator must not fit during a package/import check')

    for estimator in (IsolationForest, RandomForestClassifier, LogisticRegression, MLPClassifier):
        monkeypatch.setattr(estimator, 'fit', forbidden)
    for path in sorted((root/'src/training').glob('*.py')):
        module = importlib.import_module('src.training.'+path.stem)
        importlib.reload(module)


def test_original_model_parameters_without_fitting(settings):
    from src.training.isolation_forest import IF_PARAMS, SEEDS, BENIGN_TRAIN_FRAC
    from src.training.supervised import _models
    assert IF_PARAMS == {'n_estimators':300,'max_samples':256,'bootstrap':False}
    assert list(SEEDS) == settings['detector']['seeds']
    assert BENIGN_TRAIN_FRAC == settings['primary_reference']['fit_fraction']
    models = _models(42)
    assert set(models) == {'logistic_regression','random_forest','mlp'}
    assert models['logistic_regression'][1].get_params()['max_iter'] == 2000
    assert models['random_forest'][1].get_params()['n_estimators'] == 300
    assert models['mlp'][1].get_params()['hidden_layer_sizes'] == (64,32)


def test_training_notebook_is_unexecuted_and_opt_in(root):
    notebook = json.loads((root/'notebooks/08_train_models_in_colab.ipynb').read_text())
    code = [c for c in notebook['cells'] if c['cell_type']=='code']
    assert len(code) == 16
    for cell in code:
        assert cell['execution_count'] is None
        assert cell['outputs'] == []
        ast.parse(''.join(cell['source']))
    text = '\n'.join(''.join(c['source']) for c in code)
    for flag in ('INSTALL_TRAINING_DEPS','ALLOW_DOWNLOADS','PREPARE_INPUTS',
                 'ALLOW_TRAINING','RUN_SUPERVISED','RUN_RECORDING_PROBE'):
        assert f'{flag} = False' in text
    assert 'subprocess' not in text
    assert 'Popen' not in text
    assert notebook['metadata']['training_release']['executed'] is False


def test_optional_requirements_record_original_numeric_versions(root):
    text = (root/'requirements-training.txt').read_text()
    manifest = json.loads((root/'configs/training_source_manifest.json').read_text())
    for name, version in manifest['original_training_environment'].items():
        assert f'{name}=={version}' in text
    assert '-r requirements.txt' in text


def test_raw_inventory_is_eight_files_without_pcaps(root, tmp_path):
    workspace = TrainingWorkspace(tmp_path/'training', root)
    rows = input_inventory(workspace)
    assert len(rows) == 8
    assert not rows['path'].str.endswith('.pcap').any()
    assert not workspace.root.exists()


@pytest.mark.parametrize('kind',['sha256','git_blob_sha1'])
def test_source_hash_formats_on_small_software_fixture(tmp_path, kind):
    content = b'not a real dataset\n'
    path = tmp_path/'fixture.csv'
    path.write_bytes(content)
    expected = (hashlib.sha256(content).hexdigest() if kind=='sha256'
                else hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest())
    record = {'path':'fixture.csv','size':len(content),'sha':expected}
    verify_source(path,record)
    with pytest.raises(ValueError,match='checksum mismatch'):
        verify_source(path,{**record,'sha':'0'*len(expected)})
    with pytest.raises(ValueError,match='wrong size'):
        verify_source(path,{**record,'size':len(content)+1})


@pytest.mark.parametrize('name',['results','src','notebooks','configs','.'])
def test_workspace_rejects_canonical_repository_directories(root, name):
    with pytest.raises(ValueError):
        TrainingWorkspace(root/name,root)


def test_workspace_rejects_parent_of_repository(root):
    with pytest.raises(ValueError):
        TrainingWorkspace(root.parent,root)


def test_workspace_rejects_traversal_and_absolute_paths(root, tmp_path):
    workspace = TrainingWorkspace(tmp_path/'training',root)
    for path in ('../outside.csv',root/'results/new.csv'):
        with pytest.raises(ValueError):
            workspace.output_path(path)


def test_new_run_and_npz_writes_never_replace_files(root, tmp_path):
    workspace = TrainingWorkspace(tmp_path/'training',root)
    run1,run2 = workspace.new_run(),workspace.new_run()
    assert run1 != run2
    relative = run1/'fixture.npz'
    digest = workspace.write_npz(relative,check=np.array([1,2,3]))
    assert digest == sha256_file(workspace.output_path(relative))
    with np.load(workspace.output_path(relative)) as stored:
        np.testing.assert_array_equal(stored['check'],[1,2,3])
    assert not workspace.output_path(str(relative)+'.npz').exists()
    with pytest.raises(FileExistsError):
        workspace.write_npz(relative,check=np.array([9]))
    assert not list(workspace.root.rglob('*.partial'))


def test_model_bundle_scoring_on_stub_not_a_real_estimator():
    from src.training.model_bundle import score_bundle

    class Scaler:
        def transform(self,values):
            np.testing.assert_array_equal(values,[[1,7],[2,8]])
            return values

    class Estimator:
        def score_samples(self,values):
            return np.array([-0.2,-0.4])

    bundle = {'feature_names':['a','b'],'median':np.array([0,7]),
              'scaler':Scaler(),'model':Estimator()}
    frame = pd.DataFrame({'a':[1,2],'b':[np.nan,8]})
    np.testing.assert_array_equal(score_bundle(bundle,frame),[0.2,0.4])
    with pytest.raises(ValueError,match='names/order'):
        score_bundle(bundle,frame[['b','a']])
