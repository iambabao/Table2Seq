from .seq2seq import Seq2Seq
from .seq2seq_fa import Seq2SeqFA

model_list = {
    'seq2seq': Seq2Seq,
    'seq2seq_fa': Seq2SeqFA
}


def get_model(config, embedding_matrix=None):
    assert config.current_model in model_list

    return model_list[config.current_model](config, embedding_matrix)
