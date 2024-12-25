for id in `seq ${1} ${2}`; do
echo $id
qdel $id
done