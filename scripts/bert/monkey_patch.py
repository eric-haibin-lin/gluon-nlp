import mxnet as mx
import os
from mxnet import base, util
from mxnet.util import check_sha1, download

_model_sha1 = mxnet.gluon.model_zoo.model_store._model_sha1

def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]

def get_model_file(name, root=os.path.join(base.data_dir(), 'models')):
    r"""Return location for the pretrained on local file system.
    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.
    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    file_name = '{name}-{short_hash}'.format(name=name,
                                             short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name+'.params')
    sha1_hash = _model_sha1[name]
    import uuid
    random_uuid = str(uuid.uuid4())
    temp_root = os.path.join(root, random_uuid)
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning('Mismatch in the content of model file detected. Downloading again.')
    else:
        logging.info('Model file not found. Downloading to %s.', file_path)

    util.makedirs(root)

    temp_zip_file_path = os.path.join(root, file_name+random_uuid+'.zip')
    repo_url = os.environ.get('MXNET_GLUON_REPO', apache_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=temp_zip_file_path,
             overwrite=True)
    with zipfile.ZipFile(temp_zip_file_path) as zf:
        zf.extractall(root)
    os.remove(temp_zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')

mxnet.gluon.model_zoo.get_model_file = get_model_file
