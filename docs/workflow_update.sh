make html

git add .
git commit -m "Update workflow documentation"
git push

cp -r _build/html/* ../../Lexi-BU.github.io/

cd ../../Lexi-BU.github.io/
git add .
git commit -m "Update workflow documentation"
git push
