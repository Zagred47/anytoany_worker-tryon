'''
RunPod | ControlNet | Infer
'''

import os
import base64
import argparse
from io import BytesIO
from subprocess import call

from PIL import Image
import numpy as np

import runpod
from runpod.serverless.utils import rp_download, rp_upload
from runpod.serverless.utils.rp_validator import validate
import os
os.environ['HF_TOKEN'] = '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'

from prediction import any2any_predict
import tempfile


def get_image(image_url=None, image_base64=None):
    '''
    Get the image from the provided URL or base64 string.
    Returns a PIL image.
    '''
    if image_url is not None:
        image = rp_download.file(image_url)
        image = image['file_path']

    if image_base64 is not None:
        image_bytes = base64.b64decode(image_base64)
        image = BytesIO(image_bytes)

    input_image = Image.open(image)
    input_image = np.array(input_image)

    return input_image

def encode_img_b64(image):
    '''
    Encodes an image to base64.
    '''
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')
def predict(job):
    '''
    Run a single prediction on the model.
    '''
    job_input = job['input']

    tmp_folder = tempfile.mkdtemp()
    inputs = {}

    if job_input.get(F'src_url', None) is None and job_input.get(f'src_base64', None) is None:
        return {'error': f'No src image provided. Please provide an src_url or src_base64.'}
    elif job_input.get(f'src_url', None) is not None and job_input.get(f'src_base64', None) is not None:
        return {'error': f'Both src_url and src_base64 provided. Please provide only one.'}

    # save image in temp folder
    img_data = get_image(job_input.get(f'src_url', None), job_input.get(f'src_base64', None))
    # img_path = os.path.join(tmp_folder, f'src.png')
    #Image.fromarray(img_data).save(img_path)
    #inputs[img] = img_path
    inputs['src'] = img_data

    
    if job_input.get(F'ref_url', None) is None and job_input.get(f'ref_base64', None) is None:
        return {'error': f'No ref image provided. Please provide an ref_url or ref_base64.'}
    elif job_input.get(f'ref_url', None) is not None and job_input.get(f'ref_base64', None) is not None:
        return {'error': f'Both ref_url and ref_base64 provided. Please provide only one.'}

    # save image in temp folder
    if job_input.get('ref_url') is not None:
        source = 'image_url'
        ref_data = job_input['ref_url']
    else:
        source = 'image_base64'
        ref_data = job_input['ref_base64']

    if type(ref_data) != list:
        ref_data = [ref_data]

    ref_images = []
    height = 0
    width = 0
    for ref in enumerate(ref_data):
        img_data = get_image(**{source: ref})
        height = max(height, img_data.shape[0])
        width += img_data.shape[1]
        ref_images.append(img_data)
    
    ref_img_data = 255*np.ones((height, width), dtype=ref_images[0].dtype)
    w=0
    for img_data in ref_images:
        ref_img_data[:img_data.shape[0], w:w+img_data.shape[1]] = img_data
        w+=img_data.shape[1]

    inputs['ref'] = ref_img_data

    prompt = job_input.get('prompt', '')
    output_image = any2any_predict(inputs['src'], inputs['ref'], prompt)
    output_image = Image.fromarray(output_image.astype(np.uint8))
    return {
        'output': encode_img_b64(output_image)
    }


    # # --------------------------------- Openpose --------------------------------- #
    # elif MODEL_TYPE == "openpose":
    #     openpose_validate = validate(job_input, OPENPOSE_SCHEMA)
    #     if 'errors' in openpose_validate:
    #         return {'error': openpose_validate['errors']}
    #     validated_input = openpose_validate['validated_input']

    #     outputs = process_pose(
    #         get_image(validated_input['image_url'], validated_input['image_base64']),
    #         validated_input['prompt'],
    #         validated_input['a_prompt'],
    #         validated_input['n_prompt'],
    #         validated_input['num_samples'],
    #         validated_input['image_resolution'],
    #         validated_input['detect_resolution'],
    #         validated_input['ddim_steps'],
    #         validated_input['scale'],
    #         validated_input['seed'],
    #         validated_input['eta'],
    #         model,
    #         ddim_sampler,
    #     )

    # # outputs from list to PIL
    # outputs = [Image.fromarray(output) for output in outputs]

    # # save outputs to file
    # os.makedirs("tmp", exist_ok=True)
    # outputs = [output.save(f"tmp/output_{i}.png") for i, output in enumerate(outputs)]

    # for index, output in enumerate(outputs):
    #     outputs = rp_upload.upload_image(job['id'], f"tmp/output_{index}.png")

    # # return paths to output files
    # return outputs


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
# parser = argparse.ArgumentParser(description=__doc__)
# parser.add_argument("--model_type", type=str,
#                     default=None, help="Model URL")


if __name__ == "__main__":
    # args = parser.parse_args()

    runpod.serverless.start({"handler": predict, 'concurrency_modifier':(lambda x: 2)})
