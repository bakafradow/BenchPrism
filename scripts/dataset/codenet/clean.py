import re

import jsonlines

if __name__ == '__main__':
  src = 'data/CodeNet/result/codenet_deepseekcoder.jsonl'
  with jsonlines.open(src) as reader:
    objects = list(reader)
  for object in objects:
    # eliminate everything before the first import statement in Java
    object['result']['file_name'] = re.sub(r'^.*?(import.*)', r'\1', object['result']['file_name'], flags=re.DOTALL)
  dst = 'data/CodeNet/result/codenet_deepseekcoder_clean.jsonl'
  with jsonlines.open(dst, 'w') as writer:
    for object in objects:
      writer.write(object)
