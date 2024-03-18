# prompteval

prompteval is a tool for evaluating prompts to large language models. 


# thoughs

prompteval <model> <list of prompts> <list of testcases>

<model>
is a program that takes as input a prompt, and some user input and outputs a string.

<lists of prompts> 
is a list of prompts that a user wants to test

<list of testcases>
a testcase follows the following format

    <user input>
    user input on the format that the language model expects
    
    <ideal answer>
    is the ideal answer that the user expects out of the user input. 


