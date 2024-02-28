from transformers import RobertaTokenizer, T5ForConditionalGeneration

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

text = """
public virtual ListSpeechSynthesisTasksResponse ListSpeechSynthesisTasks(ListSpeechSynthesisTasksRequest request){var options = new InvokeOptions();options.RequestMarshaller = ListSpeechSynthesisTasksRequestMarshaller.Instance;options.ResponseUnmarshaller = ListSpeechSynthesisTasksResponseUnmarshaller.Instance;return Invoke<ListSpeechSynthesisTasksResponse>(request, options);}

"""
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate a single sequence
generated_ids = model.generate(input_ids, max_length=8)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# this prints "{user.username}"
