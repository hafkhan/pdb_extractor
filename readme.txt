please run this command in an folder with the pdb file and change the "1bey.pdb" to your pdb file name:
sudo docker run --rm -v $(pwd):/data pdb_extract python pdb_extract.py /data/1bey.pdb
