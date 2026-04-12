def token_bio_to_words(token_wid_bio):
  # [(token, word_id, bio), ...]
  # [('Me', 0, 1)]
  words = []
  pre_wid = None
  for tok, wid, bio in token_wid_bio:
    if tok is None or tok in ['[CLS]', '[PAD]', '[SEP]']:
      continue
    if bio == 0:
      words.append(tok)
    elif bio == 1 and words:
      words[-1] = words[-1] + ' ' + tok
    elif pre_wid is not None and pre_wid == wid:
      if tok.startswith('##'):
        words[-1] = words[-1] + tok[2:]
      else:
        words[-1] = words[-1] + tok
    if pre_wid != wid:
      pre_wid = wid
  return words