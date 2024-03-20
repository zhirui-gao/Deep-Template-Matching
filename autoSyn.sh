#!/bin/sh
git status  
git add *  
git commit -m 'Update readme'
git pull --rebase origin master   #domnload data
git push origin master            #upload data
git stash pop
# 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
