from .util import *
from abc import abstractmethod


class BaseModel:

    INTERLEAVE = False
    allowed_types = ['text', 'image', 'video']

    def __init__(self):
        self.dump_image_func = None

    def use_custom_prompt(self, dataset):
        return False

    @abstractmethod
    def build_prompt(self, line, dataset):
        raise NotImplementedError

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    @abstractmethod
    def generate_inner(self, message, video_start, video_end, dataset=None, answer_len=None):
        raise NotImplementedError

    def check_content(self, msgs):
        if isinstance(msgs, str):
            return 'str'
        if isinstance(msgs, dict):
            return 'dict'
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        return 'unknown'

    def preproc_content(self, inputs):
        if self.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        elif self.check_content(inputs) == 'dict':
            assert 'type' in inputs and 'value' in inputs
            return [inputs]
        elif self.check_content(inputs) == 'liststr':
            res = []
            for s in inputs:
                mime, pth = parse_file(s)
                if mime is None or mime == 'unknown':
                    res.append(dict(type='text', value=s))
                else:
                    res.append(dict(type=mime.split('/')[0], value=pth))
            return res
        elif self.check_content(inputs) == 'listdict':
            for item in inputs:
                assert 'type' in item and 'value' in item
                mime, s = parse_file(item['value'])
                if mime is None:
                    assert item['type'] == 'text'
                else:
                    assert mime.split('/')[0] == item['type']
                    item['value'] = s
            return inputs
        else:
            return None

    def generate(self, message, video_start, video_end, dataset=None, answer_len=None):
        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        message = self.preproc_content(message)
        assert message is not None and self.check_content(message) == 'listdict'
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'
        return self.generate_inner(message, video_start, video_end, dataset, answer_len)

    def chat(self, messages, dataset=None):
        assert hasattr(self, 'chat_inner'), 'The API model should has the `chat_inner` method. '
        for msg in messages:
            assert isinstance(msg, dict) and 'role' in msg and 'content' in msg, msg
            assert self.check_content(msg['content']) in ['str', 'dict', 'liststr', 'listdict'], msg
            msg['content'] = self.preproc_content(msg['content'])

        while len(messages):
            try:
                return self.chat_inner(messages, dataset=dataset)
            except Exception as e:
                print(f'{type(e)}: {e}')
                messages = messages[1:]
                while len(messages) and messages[0]['role'] != 'user':
                    messages = messages[1:]
                continue
        return 'Chat Mode: Failed with all possible conversation turns.'
