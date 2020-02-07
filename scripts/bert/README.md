## Dependency
Install gluon-nlp from source:
```
cd ../..
python3 setup.py develop --user
```
Install MXNet nightly cuxx variant from https://repo.mxnet.io/dist/index.html:
```
# Assuming cuda 10.0 below
pip3 install https://repo.mxnet.io/dist/mxnet_cu100-1.6.0-py2.py3-none-manylinux1_x86_64.whl  --user
```
Install Numpy 1.14:
```
pip3 install numpy==1.14 --user
```

## Embedding Inference
Command:
```
python3 embedding.py --gpu 0 --params_path 0300000.params.bert --sentencepiece asin-unigram-32000-150M.model --file sample_text.txt
```
Result samples:
```
Text: ▁this ▁text ▁is ▁included ▁to ▁make ▁sure ▁uni code ▁is ▁handled ▁properly ▁ : ▁ 力 ▁ 加 ▁ 勝 ▁ 北 ▁ 区 ▁ I N T a ছ জ ট ড ণ ত
Tokens embedding: [array([-0.17229098, -0.33263803,  0.2506998 , ..., -0.04724445,
       -0.4731864 ,  0.70352095], dtype=float32), array([-0.17229107, -0.33263785,  0.2506999 , ..., -0.0472445 ,
       -0.4731864 ,  0.70352083], dtype=float32), array([-0.17229116, -0.33263803,  0.25069964, ..., -0.04724413,
       -0.47318617,  0.703521  ], dtype=float32), array([-0.1722915 , -0.33263788,  0.25069982, ..., -0.0472444 ,
       -0.47318628,  0.70352095], dtype=float32), array([-0.17229138, -0.3326379 ,  0.25069964, ..., -0.04724447,
       -0.47318637,  0.7035212 ], dtype=float32), array([-0.17229152, -0.332638  ,  0.25069982, ..., -0.04724443,
       -0.473186  ,  0.7035212 ], dtype=float32), array([-0.17229117, -0.3326379 ,  0.25069985, ..., -0.0472446 ,
       -0.47318608,  0.70352083], dtype=float32), array([-0.17229134, -0.3326378 ,  0.25069982, ..., -0.04724442,
       -0.47318625,  0.7035209 ], dtype=float32), array([-0.1722912 , -0.33263746,  0.25069952, ..., -0.04724452,
       -0.47318625,  0.7035212 ], dtype=float32), array([-0.17229125, -0.33263803,  0.25069952, ..., -0.04724432,
       -0.4731866 ,  0.7035211 ], dtype=float32), array([-0.1722914 , -0.33263785,  0.2506997 , ..., -0.04724438,
       -0.47318637,  0.703521  ], dtype=float32), array([-0.17229116, -0.3326379 ,  0.25069964, ..., -0.0472444 ,
       -0.4731863 ,  0.703521  ], dtype=float32), array([-0.17229122, -0.33263803,  0.25069964, ..., -0.04724439,
       -0.4731862 ,  0.70352125], dtype=float32), array([-0.17229113, -0.3326379 ,  0.25069958, ..., -0.04724447,
       -0.4731861 ,  0.70352113], dtype=float32), array([-0.17229116, -0.33263785,  0.25069955, ..., -0.04724438,
       -0.4731865 ,  0.70352125], dtype=float32), array([-0.17229143, -0.33263808,  0.2506997 , ..., -0.0472444 ,
       -0.4731862 ,  0.703521  ], dtype=float32), array([-0.17229117, -0.3326379 ,  0.25069982, ..., -0.04724456,
       -0.47318608,  0.70352125], dtype=float32), array([-0.17229128, -0.33263797,  0.25069964, ..., -0.04724437,
       -0.47318628,  0.70352113], dtype=float32)]
...
```
