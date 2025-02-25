import sys
import json

import torch
import esm
import time

from datetime import datetime
import os
import torch.nn.functional as F


def validate_pdb(pdb_file_path):
    try:
        with open(pdb_file_path, 'r') as pdb_file:
            required_atoms = {"C", "CA", "O", "N"}
            requiredAtomsFlag = [False,False,False,False]
            found_atoms = set()
            idx = 0

            for line in pdb_file:
                idx = idx + 1
                # print("---------- lines  : " + str(idx) + " " + line)

                if line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    # print("---------- atom name : " + atom_name)
                    found_atoms.add(atom_name)
                    if(atom_name == "C"):
                        requiredAtomsFlag[0] = True
                    elif(atom_name == "CA"):
                        requiredAtomsFlag[1] = True
                    elif(atom_name == "O"):
                        requiredAtomsFlag[2] = True
                    elif(atom_name == "N"):
                        requiredAtomsFlag[3] = True
                    if (requiredAtomsFlag[0] & requiredAtomsFlag[1] &
                        requiredAtomsFlag[2] & requiredAtomsFlag[3]):
                        return True  # Early exit if all required atoms are found

            missing_atoms = required_atoms - found_atoms
            if missing_atoms:
                raise ValueError(f"Missing required atom types: {', '.join(missing_atoms)}")
    except FileNotFoundError:
        print("Error: PDB file not found.")
    except Exception as e:
        print(f"Error: {e}")

    return False

def mapping_extract(val):
    with open("mapping.json", 'r') as mapfile:
        data = json.load(mapfile)
        return data.get(val, "Key not Found")
def return_chain(chainValues, chainID):
    lastElement = len(chainValues) - 1
    chainValues[lastElement] = mapping_extract(chainValues[lastElement])
    retVal = {}
    retVal['length'] = len(chainValues)
    retVal['chainID'] = chainID
    retVal['chain'] = ["".join(chainValues)]
    return retVal

def chain_element_check(chainValues, newVal):
    lastElement = len(chainValues) - 1
    if(chainValues[lastElement] != newVal):
        retChain = mapping_extract(chainValues[lastElement])
        if(retChain == "Key not Found"):
            return False
        chainValues[lastElement] = retChain
        chainValues.append(newVal)
    return True
    
def chain_extractor(pdb_file_path):
    try:
        with open(pdb_file_path, 'r') as pdb_file:
            state = "init"
            chainValues = []
            chains = {}
            chainLength = 0
            #print("---------- start extractiong\n")
            for line in pdb_file:
                # print("---------- lines  : ")
                if(line.startswith("ATOM")):
                    chId = line[21]
                    newVal = line[17:20].strip()
                    if(chId == "H" or chId == "L"):
                        if(state == "init"):
                            # new chain. let's start the chain
                            state = chId
                            chainValues.append(newVal)
                        elif(state == "H"):
                            if(chId == "H"):
                                # we are already in a chain! check the chainValues for duplication.
                                if(not chain_element_check(chainValues=chainValues, newVal= newVal)):
                                    raise TypeError()
                            elif(chId == "L"):
                                # chain is changed. we have to change the state and the chains and
                                # add selected chain to the chain dictionary.
                                chains[chainLength] = return_chain(chainValues=chainValues, chainID=state)
                                chainLength = chainLength + 1
                                # start new state
                                chainValues.clear()
                                state = chId
                                chainValues.append(newVal)
                        elif(state == "L"):
                            if(chId == "H"):
                                # chain is changed. we have to save previous chain and
                                # start a new state and chain
                                chains[chainLength] = return_chain(chainValues=chainValues, chainID=state)
                                chainLength = chainLength + 1
                                # start new state
                                chainValues.clear()
                                state = chId
                                chainValues.append(newVal)
                            elif(chId == "L"):
                                # we are already in a chain! check the chainValues for duplication.
                                if(not chain_element_check(chainValues=chainValues, newVal= newVal)):
                                    raise TypeError("")
                    else:
                        if(state == "H" or state == "L"):
                            # chain is changed. we have to change the state and the chains and
                            # add selected chain to the chain dictionary.
                            chains[chainLength] = return_chain(chainValues=chainValues, chainID=state)
                            chainLength = chainLength + 1
                            # start new state
                            chainValues.clear()
                            state = "init"
                elif(line.startswith("TER")):
                    #armin = 2
                    if(state == "H" or state == "L"):
                        chains[chainLength] = return_chain(chainValues=chainValues, chainID=state)
                        chainLength = chainLength + 1
                        # start new state
                        chainValues.clear()
                        state = "init"
            # end of the file. Let's extract the last chain which we were into 
            if(state != "init"):
                chains[chainLength] = return_chain(chainValues=chainValues, chainID=state)
            return chains
    except TypeError:
        print("Error: Map key not found")
        return {}
    except FileNotFoundError:
        print("Error: PDB file not found.")
        return {}
    except Exception as e:
        print(f"Error: {e}")
        return {}
    return {}

def ESM_model(chains):
    try:
        # Load ESM-2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results

        data = []
        for chn in chains:
            data.append(("protein " + chains[chn]['chainID'], chains[chn]['chain'][0]))

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        startTime = time.time()
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)

        endTime = time.time()
        prediction_time = endTime - startTime
        print("Time : " + str(prediction_time))
        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

        # Extract logits from the model output
        logits = results["logits"]

        # Convert logits to class probabilities using softmax
        probabilities = F.softmax(logits, dim=-1)  # Convert logits to probabilities
        predicted_class_ids = torch.argmax(probabilities, dim=-1)  # Get the most probable class index

        output_metadata = []
        for i, (label, seq) in enumerate(data):
            output_metadata.append({
                "sequence_label": label,  # Name of the sequence (e.g., "protein L")
                "prediction_time": prediction_time,  # Same for all sequences (or measure separately per sequence)
                "original_input_sequence": seq,  # Raw sequence
                "input_tokens": batch_tokens[i].tolist(),  # Tokenized version of the sequence
                "model_output_tensor": sequence_representations[i].tolist(),  # Per-sequence embedding
                "predicted_result": predicted_class_ids[i].tolist()  # Per-sequence predicted class IDs
            })

        # Detect if running inside Docker
        if os.path.exists("/.dockerenv"):
            OUTPUT_DIR = "/data"  # Docker: Write to /data (mounted to the host)
        else:
            OUTPUT_DIR = os.getcwd()  # Local: Write in the current working directory

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        outputFilelearning = os.path.join(OUTPUT_DIR, "learning_output_metadata.json")
        # Save the output to a JSON file
        with open(outputFilelearning, "w") as f:
            json.dump(output_metadata, f, indent=4)

        print("learning output saved to learning_output_metadata.json")
    except Exception as e:
        print(f"Error in ESM model: {e}")

def process_file(pdb_file_path):
    #pdb_file_path = "test.pdb"
    try:
        valid = validate_pdb(pdb_file_path)
        if(valid):
            print("PDB file is valid.")
            chains = chain_extractor(pdb_file_path)
            # 4. File Format and Metadata
            # • Include the following metadata in the output file: 
            # ◦ Protein ID or filename
            # ◦ Date and time of processing
            # ◦ Extracted chains (H, L) and its length

            currentTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            chainsList = [chains[idx] for idx in chains]  # Extract values while keeping order

            # Construct the output JSON structure
            outputFile = {
                "filename": pdb_file_path,
                "timestamp": currentTime,
                "chains": chainsList  # Store as a list
            }
            # Detect if running inside Docker
            if os.path.exists("/.dockerenv"):
                OUTPUT_DIR = "/data"  # Docker: Write to /data (mounted to the host)
            else:
                OUTPUT_DIR = os.getcwd()  # Local: Write in the current working directory

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            outputFileChain = os.path.join(OUTPUT_DIR, "chain_output_metadata.json")
            # Save to JSON file
            with open(outputFileChain, "w") as f:
                json.dump(outputFile, f, indent=4)
            print("Chains are save in output file.");
            print("Start applying ESM Model.")
            if(chains != {}):
                ESM_model(chains)
        else:
            print("Error in Validation")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdb_extract.py <file_path>")
        exit(1)

    input_file = sys.argv[1]
    # input_file = "test.pdb"
    process_file(input_file)