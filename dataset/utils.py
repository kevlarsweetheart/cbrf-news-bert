import json
import gzip
import io


def load_lines(file_path, _encoding='utf-8'):
    with open(file_path, encoding=_encoding) as infile:
        for line in infile:
            yield line.rstrip('\n')


def dump_lines(lines, file_path, _encoding='utf-8'):
    with open(file_path, 'w', encoding=_encoding) as outfile:
        for line in lines:
            outfile.write(line + '\n')


def load_gz_lines(file_path, _encoding='utf-8'):
    with gzip.open(file_path) as _file:
        infile = io.BufferedReader(_file)
        for line in infile:
            yield line.decode(_encoding).rstrip()


def dump_gz_lines(lines, file_path):
    with gzip.open(file_path, 'wt') as outfile:
        try:
            outfile.writelines(lines)
        finally:
            outfile.close()


def load_json(file_path, _encoding='utf-8'):
    with open(file_path, encoding=_encoding) as infile:
        return json.load(infile)


def serialize_json(items):
    for item in items:
        yield json.dumps(item, ensure_ascii=False)


def deserialize_json(lines):
    for line in lines:
        yield json.loads(line)
