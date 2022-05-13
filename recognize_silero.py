from glob import glob
from silero import silero_stt, silero_tts, silero_te

if __name__ == "__main__":
    # after
    model, decoder, utils = silero_stt()
    (read_batch, split_into_batches,
     read_audio, prepare_model_input) = utils  # see function signature for details
    device = 'cpu'
    # download a single file, any format compatible with TorchAudio
    # torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
    #                                dst='speech_orig.wav', progress=True)
    test_files = glob("/home/user/projects/Subtitles-generator/samples/*wav")
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]),
                                device=device)

    output = model(input)
    for example in output:
        print(decoder(example.cpu()))

