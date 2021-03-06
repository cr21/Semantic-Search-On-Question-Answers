import tensorflow as tf
import tensorflow_hub as hub

model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embeddings = tf.make_ndarray(
                tf.make_tensor_proto(
                    model(
                        ["The quick brown fox jumps over the lazy dog."]
                    )
                )
            ).tolist()[0]

print(type(embeddings))
print(len(embeddings))
print(embeddings)
