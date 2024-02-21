from pdf2image import convert_from_path

file_location = "/home/undergrad/Downloads/theta_search_n=50.pdf"
output_location = file_location.replace(".pdf", ".png")

image = convert_from_path(file_location, dpi=600)[0]

print(image, type(image))

# save PIL.PpmImagePlugin.PpmImageFile to the disk
image.save(output_location)