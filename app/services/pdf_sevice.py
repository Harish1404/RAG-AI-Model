from unstructured.partition.pdf import partition_pdf


def extract_elements_from_pdf(file_path: str):
    """
    Extract structured elements from PDF
    """

    elements = partition_pdf(file_path)

    texts = []

    for element in elements:
        texts.append(str(element))

    return texts