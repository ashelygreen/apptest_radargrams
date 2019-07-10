import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

# import cv2
# import base64
from includes import *

export_file_url = 'https://www.dropbox.com/s/8nf7v81qsrxgv8y/trained_model_resnet152.pkl?dl=1'
export_file_name = 'trained_model_resnet152.pkl'

classes = ['Grave', 'Other']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        learn.loss_func = MyLoss()
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n"
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())
 
@app.route('/favicon.ico', methods=["GET"])
def form(request):
    return RedirectResponse(url='/static/favicon.ico')
 
@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
 
    prediction = predict(learn, img)
    # arr = (img.data.numpy().transpose(1,2,0) * 255).astype('uint8')   
     
    # if prediction[1][1] > 0.5:
    #     bbox, text = show(prediction[0], noshow=True)
    #     arr = cv2.rectangle(arr.copy(), tuple(bbox[:2]), tuple(bbox[:2]+bbox[2:]), (255, 0, 0), thickness=1)
    # else:
    #     arr = cv2.putText(arr.copy(), 'No detection', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (252, 0, 0), thickness=2)
         
    # retval, buffer = cv2.imencode('.png', arr[:,:,::-1])
    # img_as_text = base64.b64encode(buffer)
    # return PlainTextResponse('data:image/png;base64,%s' % img_as_text.decode())
     
    bbox, text = show(prediction[0], noshow=True)
    out = {
        'prob': str(prediction[1][1].numpy()[0]), 
        'bbox': str(bbox.astype('int').tolist()),
        'size': str(list(img.shape[1:]))
    }
    return JSONResponse(out)
 
 
if __name__ == '__main__':
    if 'serve' in sys.argv or True:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")