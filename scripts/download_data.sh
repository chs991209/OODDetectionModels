#!/bin/bash

# 데이터 디렉토리 생성
mkdir -p ./data/animals
mkdir -p ./data/pokemon

# --- 1. [ID] Animals-10 (수동 구축된 데이터 보호) ---
# data/animals 안에 폴더(dog, cat 등)가 하나라도 있는지 확인
if [ -d "./data/animals/dog" ] || [ -d "./data/animals/cat" ]; then
    echo ">>> [Animal] 데이터가 이미 존재합니다. (Skipping download)"
else
    echo ">>> [Animal] 경고: Animals-10 데이터가 없습니다!"
    echo "    Dropbox 링크가 만료되었으므로, Kaggle에서 수동으로 다운로드하여 ./data/animals 경로에 넣어주세요."
    echo "    (현재 상태에서는 학습 스크립트 실행 시 에러가 발생할 수 있습니다.)"
fi

# --- 2. [OOD] Pokemon (PokeAPI -> 정제) ---
# data/pokemon/unknown 폴더에 이미지가 있는지 확인
if [ -z "$(ls -A ./data/pokemon/unknown 2>/dev/null)" ]; then
    echo ">>> [Pokemon] PokeAPI 공식 리포지토리 클론 및 데이터 생성 시작..."

    # 임시 폴더 청소
    rm -rf ./data/pokeapi_temp

    # 1. 공식 리포지토리 Clone (Depth 1로 빠르게)
    git clone --depth 1 https://github.com/PokeAPI/sprites.git ./data/pokeapi_temp

    target_dir="./data/pokemon/unknown"
    mkdir -p "$target_dir"

    echo ">>> [Pokemon] 데이터 검증 및 변환 (Python)..."
    python3 -c "
import os
from PIL import Image
from tqdm import tqdm

# 경로 설정
source = './data/pokeapi_temp/sprites/pokemon/other/official-artwork'
target = '$target_dir'

if not os.path.exists(source):
    print(f'Error: 소스 경로를 찾을 수 없습니다: {source}')
    exit(1)

files = [f for f in os.listdir(source) if f.endswith('.png')]
print(f'>>> 총 {len(files)}장의 포켓몬 이미지를 발견했습니다.')

count = 0
for f in tqdm(files):
    try:
        with Image.open(os.path.join(source, f)) as img:
            if getattr(img, 'is_animated', False): continue # GIF 제거

            # 투명 배경(RGBA) -> 흰색 배경 합성
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                final = bg
            else:
                final = img.convert('RGB')

            final.save(os.path.join(target, f.replace('.png', '.jpg')), quality=95)
            count += 1
    except: pass

if count < 100:
    print('Error: 변환된 이미지가 너무 적습니다.')
    exit(1)
print(f'>>> 변환 완료: {count}장 저장됨.')
"

    # Python 스크립트 성공 여부 확인
    if [ $? -eq 0 ]; then
        echo ">>> [Pokemon] 데이터 준비 완료."
        rm -rf ./data/pokeapi_temp
    else
        echo ">>> [Pokemon] 데이터 생성 실패!"
        # 실패 시 임시 파일 남겨둠 (디버깅용)
    fi

else
    echo ">>> [Pokemon] 데이터가 이미 존재합니다. (Skipping download)"
fi