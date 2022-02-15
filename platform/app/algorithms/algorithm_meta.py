

class Algorithm_Meta(type):
    """ Algorithm meta class """

    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        return (hasattr(subclass, 'train_algorithm_unsupervised') and
                callable(subclass.train_algorithm_unsupervised) and
                hasattr(subclass, 'predict_sample') and
                callable(subclass.predict_sample) and
                hasattr(subclass, 'load_model_from_file') and
                callable(subclass.load_model_from_file) and
                hasattr(subclass, 'store_model_to_file') and
                callable(subclass.store_model_to_file) and
                hasattr(subclass, 'store_model_to_s3') and
                callable(subclass.store_model_to_s3) and
                hasattr(subclass, 'load_model_from_s3') and
                callable(subclass.load_model_from_s3))



