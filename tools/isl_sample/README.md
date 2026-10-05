# The ISL sample clip

`web/samples/isl_hello_how_are_you_thank_you.mp4` is **real**: one Deaf signer
from the [INCLUDE](https://zenodo.org/records/4010759) dataset signing three
Indian Sign Language words, "Hello", "How are you", "Thank you".

- INCLUDE: Sridhar, Ganesan, Kumar and Khapra, *INCLUDE: A Large Scale Dataset
  for Indian Sign Language Recognition*, ACM Multimedia 2020. Recorded with
  Deaf students of St. Louis School for the Deaf, Adyar, Chennai. Licence
  CC BY 4.0, which allows sharing with credit; the credit is also burned into
  the clip.
- The three clips (`Greetings/48. Hello/MVI_0029`, `49. How are you/MVI_0034`,
  `55. Thank you/MVI_0061`) are among the 5 per word held out before training
  (`results/words.json`, `held_out_clips`): the model never saw them.
- `make_sample.py` crops each clip around the upper body (webcam-like
  framing, from the landmarks), scales it to 640x480 and joins them. The
  signer rests with hands down between words, which is how the app splits
  signs.

Why a real clip here, when the letter sample is synthetic: a fingerspelled
letter is a still hand shape that can be posed and checked. Word signs are
movements of a real language; animating them myself would mean inventing them.
