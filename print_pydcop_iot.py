import inspect
import pydcop.commands.generators.iot as iot
print('FILE:', iot.__file__)
print('\n---SOURCE START---\n')
print(inspect.getsource(iot))
print('\n---SOURCE END---\n')
