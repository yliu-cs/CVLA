# for modal in "title" "comment" "visual" "acoustic" "title_comment" "title_visual" "title_acoustic" "comment_visual" "comment_acoustic" "visual_acoustic" "title_comment_visual" "title_comment_acoustic" "title_visual_acoustic" "comment_visual_acoustic" "title_comment_visual_acoustic"
for modal in "text_video"
do
    for seed in 2 42 327 2023 998244353
    do
        python run.py \
            --modal=$modal \
            --seed=$seed
    done
done

python tools/gather_result.py > gathered_result.log
