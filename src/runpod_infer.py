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


def get_image(image_url, image_base64):
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
    for img in ['src', 'ref']:
        if job_input.get(F'{img}_url', None) is None and job_input.get(f'{img}_base64', None) is None:
            return {'error': f'No {img} image provided. Please provide an {img}_url or {img}_base64.'}
        elif job_input.get(f'{img}_url', None) is not None and job_input.get(f'{img}_base64', None) is not None:
            return {'error': f'Both {img}_url and {img}_base64 provided. Please provide only one.'}
        
        # save image in temp folder
        img_data = get_image(job_input.get(f'{img}_url', None), job_input.get(f'{img}_base64', None))
        img_path = os.path.join(tmp_folder, f'{img}.png')
        Image.fromarray(img_data).save(img_path)
        inputs[img] = img_path
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
