import json

# convert dict or list to pretty-printed string
def ensure_string(value):
    if isinstance(value, (dict, list)):
        # Pretty-print JSON as string
        return json.dumps(value, indent=4)
    elif isinstance(value, str):
        return value
    else:
        raise TypeError(f"Unsupported type: {type(value)}")

# load json file
#file = './fewshots/coordinator_handoff.json'
#with open(file, 'r') as f:
#    data = json.load(f)

#f = ensure_string(data[0]['output'])
#print(f'Final Output: {f}')  # Pretty-printed JSON