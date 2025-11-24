import inspect
import logging
logger = logging.getLogger(__name__)

@staticmethod
def convert(img, module):
    if isinstance(img, str):
        return img
    if module == None:
        return img
    t = type(img)
    if inspect.getmodule(t) == module:
        return img
    if logging.DEBUG >= logging.root.level:
        finfo = inspect.getouterframes(inspect.currentframe())[1]
        logger.log(logging.DEBUG,
                   F'convert {t.__module__} to {module.__name__} by {finfo.filename} line {finfo.lineno}')
    if t.__module__ == 'cupy':
        return module.array(img.get(), copy=False)
    return module.array(img, copy=False)