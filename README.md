# ldm_poster
## scripts 파일 setting

```
LATENT-DIFFUSION
├── ldm
│   └── models
│        └── diffusion
│            ├── ddim.py
│            └── ddpm.py
└── scripts
    ├── latent_imagenet_diffusion.ipynb
    └── latent_imagenet_diffusion.py

```
- github link: CompVis/latent-diffusion: High-Resolution Image Synthesis with Latent Diffusion Models (github.com)
- 위 링크에서 먼저 git clone 받고, 필요한 가상환경 설치하기
- github에 올라간 scripts 폴더 안에 4개의 파일이 있는데, 다운받은 다음 위 **디렉토리 구조에 맞게** 원래 파일을 삭제하고 해당 파일로 대체하기
- latent_imagenet_diffusion.ipynb: **자기 경로에 맞게 수정**해야 함, **실질적으로 실행하는 파일**
- latent_imagenet_diffusion.py: 필수 아님, 실행하려면 **자기 경로에 맞게 수정**해야 함
  - 이 파일의 용도는 디버깅 용입니다. (ipynb 파일은 디버깅이 안됨) 예를 들어서 ddim.py나 ddpm.py 안에 있는 코드를 한 줄씩 디버깅하려면 필요하다.  
- 에러 뜨면 말씀해주셔요 ㅠㅠ
