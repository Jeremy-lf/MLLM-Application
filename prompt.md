[ALLaVA-Source From ALLaVa](https://github.com/FreedomIntelligence/ALLaVA/tree/main?tab=readme-ov-file) and 
[LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main)

## Instruction for caption

```
Write a comprehensive caption for the image provided.
Can you help me understand the image by providing a detailed caption?
Please provide a vivid description of the image.
Elaborate on the details of the image provided.
Could you please interpret the image and write a detailed caption?
Please depict the image in words.
How would you describe the image to someone who cannot see it?
Please enlighten me with a detailed description of the image.
Can you transform the visual elements of the image into words?
Please provide a detailed written representation of the image.
Could you please transcribe the image into a descriptive paragraph?
Please illustrate the image through your words.
Please provide a detailed narrative of the image.
Could you please express the image in a descriptive format?
Please convert the visual information in the image into a detailed written explanation.

```

## prompt_for_laion
```
### You are an excellent image describer and questioner
### You have three tasks in total
#### Your first task is to describe the given image as detailed as possible
#### Your second task is to ask a complex question that requires close inspection of the image and strong reasoning ability to answer, you should ask FIVE candidate questions in different aspects and diverse ways, then RANDOMLY choose one of them to answer
#### Your third task is to answer the question you raised solely based on the given image
### When you ask questions, try to find the most valuable information in the picture to ask about, and ask a question that is relevant to that information
### When you ask questions, do not involve violence, advertisement, possible invasion of privacy, or questions that may cause discomfort
### Do not mention anything from the prompt in your response
### You will follow the instructions to the best of your ability
### Your response should follow the following format
<start of description>
{description}
<end of description>
<start of candidate questions>
{candidate questions}
<end of candidate questions>
<start of question>
{question}
<end of question>
<start of answer>
{answer}
<end of answer>
```
<img width="1288" height="1090" alt="image" src="https://github.com/user-attachments/assets/6bda5f35-62d6-417c-9388-b3ff924e94ef" />


```
{
  "id": "allava_laion_cap_100277305",
  "image": "allava_laion/images/100277305.jpeg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nPlease depict the image in words."
    },
    {
      "from": "gpt",
      "value": "The image displays a silver ring with a design that resembles a stylized leaf. The leaf design is split into two symmetrical halves, separated by a gap that runs along the ring's band. Each half of the leaf pattern is textured with detailed, vein-like striations that give it a realistic look. The edges of the leaf design are embellished with small, clear gemstones that could possibly be diamonds, adding a sparkle to the ring. The band of the ring is wide, supporting the large leaf motif, and it tapers slightly as it moves away from the leaf design. The silver metal has a high polish, reflecting light brightly and providing a contrast to the textured leaf surface. The craftsmanship suggests a mix of organic inspiration with a touch of elegance due to the gems
tone accents."
    }
  ],
  "original_caption": "190952cz Pandora Tropical Palm Leaf Ring",
  "url": "https://theartofpandora.com/wp-content/uploads/2016/10/190952CZ-1-600x600.jpg"
}
```

```
{
  "id": "allava_vflan_cap_100000",
  "image": "allava_vflan/images/images_191task_1k/PACS+guitar_image_category_classification_830_n03272010_10795-5.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nHow would you describe the image to someone who cannot see it?"
    },
    {
      "from": "gpt",
      "value": "The image shows a simplistic line drawing of an electric guitar. The guitar is oriented diagonally from the top left corner to the bottom right corner of the image. It has a long, narrow neck with tuning pegs at the top, resembling a stylized \"T\" shape. The body of the guitar is irregular, with two cutaways on either side of the neck, giving it an asymmetric appearance. The body is also characterized by sharp angles rather than curves, which is not typical for most guitars. There are four horizontal lines across the body, likely representing strings. The drawing is black and white, with no shading or color, and appears to be hand-drawn with a single continuous line without much detail. The background is plain and white."
    }
  ]
}

```
